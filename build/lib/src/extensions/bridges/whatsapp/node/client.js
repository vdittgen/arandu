/**
 * Custom Baileys WhatsApp client for Arandu.
 *
 * Communicates with the Python listener via JSONL over stdio:
 *   stdout → events/responses (Python reads)
 *   stdin  ← commands (Python writes)
 *   stderr → Baileys/pino logs (goes to log file)
 *
 * Usage:
 *   node client.js --auth-dir ~/.claude/connectors/whatsapp/auth
 */

import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  downloadMediaMessage,
} from "@whiskeysockets/baileys";
import pino from "pino";
import fs from "fs";
import path from "path";
import readline from "readline";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const AUTH_DIR = (() => {
  const idx = process.argv.indexOf("--auth-dir");
  if (idx !== -1 && process.argv[idx + 1]) return process.argv[idx + 1];
  const eq = process.argv.find((a) => a.startsWith("--auth-dir="));
  if (eq) return eq.split("=")[1];
  return path.join(process.env.HOME || "~", ".claude", "connectors", "whatsapp", "auth");
})();

const STORE_PATH = path.join(AUTH_DIR, "store.json");
const AUDIO_CACHE_DIR = path.join(
  process.env.HOME || "~",
  ".arandu",
  "data",
  "audio_cache",
);
const STORE_SAVE_INTERVAL_MS = 10_000;
const SELF_CHAT_POLL_INTERVAL_MS = 60_000;
const MAX_MESSAGES_PER_CHAT = 200;
const AUDIO_DOWNLOAD_TIMEOUT_MS = 30_000;

// Baileys logs go to stderr so stdout stays clean for JSONL protocol.
const logger = pino({ level: "warn" }, pino.destination(2));

// ---------------------------------------------------------------------------
// JSONL helpers — stdout is our protocol channel
// ---------------------------------------------------------------------------

function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function emitResponse(id, ok, data) {
  emit(ok ? { type: "response", id, ok: true, data } : { type: "error", id, error: data });
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/** Convert Baileys protobuf Long / number / string to unix seconds. */
function toEpoch(ts) {
  if (ts == null) return Math.floor(Date.now() / 1000);
  if (typeof ts === "object" && "low" in ts) return ts.low >>> 0;
  const n = Number(ts);
  // If > 1e12 it's probably milliseconds
  return n > 1e12 ? Math.floor(n / 1000) : n;
}

// ---------------------------------------------------------------------------
// Message text extraction (mirrors Python _extract_whatsapp_message_text)
// ---------------------------------------------------------------------------

function extractText(msg) {
  if (!msg) return null;
  return (
    msg.conversation ||
    msg.extendedTextMessage?.text ||
    msg.imageMessage?.caption ||
    msg.videoMessage?.caption ||
    msg.documentMessage?.caption ||
    msg.documentMessage?.fileName ||
    null
  );
}

/** Check if a message contains an audio/voice note. */
function isAudioMessage(msg) {
  return msg?.audioMessage != null;
}

/** Get audio metadata from a message. */
function getAudioMeta(msg) {
  const audio = msg?.audioMessage;
  if (!audio) return null;
  return {
    seconds: audio.seconds || 0,
    mimetype: audio.mimetype || "audio/ogg; codecs=opus",
    ptt: Boolean(audio.ptt), // push-to-talk = voice note
    file_length: audio.fileLength ? Number(audio.fileLength) : 0,
  };
}

/**
 * Download audio from a WhatsApp message and save to audio cache.
 * Returns the file path on success, null on failure.
 */
async function downloadAudio(fullMsg) {
  const msgId = fullMsg.key?.id;
  if (!msgId) return null;

  const outPath = path.join(AUDIO_CACHE_DIR, `${msgId}.ogg`);

  // Skip if already downloaded
  if (fs.existsSync(outPath)) return outPath;

  try {
    fs.mkdirSync(AUDIO_CACHE_DIR, { recursive: true });

    const buffer = await Promise.race([
      downloadMediaMessage(fullMsg, "buffer", {}),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error("Audio download timed out")),
          AUDIO_DOWNLOAD_TIMEOUT_MS,
        ),
      ),
    ]);

    // Validate buffer (known Baileys bug: empty buffers)
    if (!buffer || buffer.length === 0) {
      process.stderr.write(`Empty audio buffer for msg ${msgId}\n`);
      return null;
    }

    fs.writeFileSync(outPath, buffer);

    // Write sidecar metadata for Python transcription matching
    const metaPath = outPath.replace(/\.ogg$/, ".json");
    const jid = fullMsg.key?.remoteJid || "";
    fs.writeFileSync(
      metaPath,
      JSON.stringify({
        msg_id: msgId,
        jid,
        full_id: `${jid}:${msgId}`,
        from_me: Boolean(fullMsg.key?.fromMe),
        timestamp: fullMsg.messageTimestamp
          ? Number(fullMsg.messageTimestamp)
          : null,
      }),
    );
    return outPath;
  } catch (err) {
    process.stderr.write(
      `Audio download failed for ${msgId}: ${err.message}\n`,
    );
    return null;
  }
}

// ---------------------------------------------------------------------------
// JID helpers
// ---------------------------------------------------------------------------

/** Strip device suffix and domain → bare number. */
function bareJid(jid) {
  if (!jid) return "";
  return jid.split(":")[0].split("@")[0];
}

// ---------------------------------------------------------------------------
// In-memory store with file persistence
// ---------------------------------------------------------------------------

const store = {
  chats: {},
  contacts: {},
  messages: {},
};

function loadStore() {
  try {
    if (fs.existsSync(STORE_PATH)) {
      const data = JSON.parse(fs.readFileSync(STORE_PATH, "utf-8"));
      if (data.chats) store.chats = data.chats;
      if (data.contacts) store.contacts = data.contacts;
      if (data.messages) store.messages = data.messages;
    }
  } catch {
    // Start fresh on corrupted store
  }
}

function saveStore() {
  try {
    const tmp = STORE_PATH + ".tmp";
    fs.writeFileSync(tmp, JSON.stringify(store, null, 2), "utf-8");
    fs.renameSync(tmp, STORE_PATH);
    emit({ type: "store_updated" });
  } catch (err) {
    process.stderr.write(`Store save error: ${err.message}\n`);
  }
}

/** Append a message to the store, capping per-chat history. */
function storeMessage(jid, msg) {
  if (!store.messages[jid]) store.messages[jid] = [];
  const arr = store.messages[jid];
  // Dedup by message ID
  if (arr.some((m) => m.key?.id === msg.key?.id)) return;
  arr.push(msg);
  if (arr.length > MAX_MESSAGES_PER_CHAT) {
    store.messages[jid] = arr.slice(-MAX_MESSAGES_PER_CHAT);
  }
}

// ---------------------------------------------------------------------------
// Self-chat detection
// ---------------------------------------------------------------------------

let myBareJid = "";
let myBareLid = "";
const emittedSelfChatIds = new Set();

function isSelfChat(remoteJid) {
  if (!remoteJid) return false;
  const bare = bareJid(remoteJid);
  // Match against phone JID (@s.whatsapp.net)
  if (myBareJid && bare === myBareJid) return true;
  // Match against linked-device JID (@lid) — self-chat messages from the
  // phone arrive under @lid JIDs in multi-device Baileys.
  if (myBareLid && bare === myBareLid) return true;
  return false;
}

function emitSelfChatMessage(msg) {
  const msgId = msg.key?.id;
  if (!msgId || emittedSelfChatIds.has(msgId)) return;
  const text = extractText(msg.message);

  // Handle audio messages: download + emit even without text
  if (!text && isAudioMessage(msg.message)) {
    emittedSelfChatIds.add(msgId);
    // Download audio in background — Python will transcribe it
    downloadAudio(msg)
      .then((audioPath) => {
        if (audioPath) {
          const audioMeta = getAudioMeta(msg.message);
          emit({
            type: "audio_downloaded",
            jid: msg.key?.remoteJid || "",
            msg_id: msgId,
            from_me: Boolean(msg.key?.fromMe),
            audio_path: audioPath,
            duration_seconds: audioMeta?.seconds || 0,
            ptt: audioMeta?.ptt || false,
            timestamp: toEpoch(msg.messageTimestamp),
          });
        }
      })
      .catch((err) => {
        process.stderr.write(
          `Self-chat audio download failed for ${msgId}: ${err.message}\n`,
        );
      });
    return;
  }

  if (!text) return;

  emittedSelfChatIds.add(msgId);

  // Extract quoted message text for thread replies
  const msgBody = msg.message || {};
  const ctxInfo =
    msgBody.extendedTextMessage?.contextInfo ||
    msgBody.contextInfo ||
    {};
  const quotedMsg = ctxInfo.quotedMessage || {};
  const quotedText =
    quotedMsg.conversation ||
    quotedMsg.extendedTextMessage?.text ||
    "";

  emit({
    type: "self_chat",
    msg_id: msgId,
    from_me: Boolean(msg.key?.fromMe),
    text,
    timestamp: toEpoch(msg.messageTimestamp),
    quoted_text: quotedText || undefined,
  });
}

// ---------------------------------------------------------------------------
// Socket lifecycle
// ---------------------------------------------------------------------------

let sock = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 20;
let storeInterval = null;
let selfChatPollInterval = null;
let shuttingDown = false;

/** Wipe stored credentials/session keys after a logged-out disconnect,
 * keeping `store.json` (local message history) intact. */
function clearAuthState() {
  try {
    for (const entry of fs.readdirSync(AUTH_DIR)) {
      if (entry === "store.json") continue;
      fs.rmSync(path.join(AUTH_DIR, entry), { recursive: true, force: true });
    }
  } catch (err) {
    process.stderr.write(`Failed to clear stale auth state: ${err.message}\n`);
  }
}

async function connectSocket() {
  if (shuttingDown) return;

  fs.mkdirSync(AUTH_DIR, { recursive: true });

  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    version,
    logger,
    printQRInTerminal: false,
    browser: ["Arandu", "Desktop", "1.0"],
    syncFullHistory: true,
    markOnlineOnConnect: false,
    emitOwnEvents: true,
  });

  // --- Credential persistence ---
  sock.ev.on("creds.update", saveCreds);

  // --- Connection updates ---
  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      emit({ type: "qr", qr });
    }

    if (connection === "open") {
      reconnectAttempts = 0;
      myBareJid = bareJid(sock.user?.id || "");
      myBareLid = bareJid(sock.user?.lid || "");
      emit({ type: "ready", jid: myBareJid, lid: myBareLid || null });
      emit({ type: "connection", status: "open" });
      startPeriodicTasks();
    }

    if (connection === "close") {
      stopPeriodicTasks();
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const reason = lastDisconnect?.error?.message || `status ${statusCode}`;
      emit({ type: "connection", status: "close", reason, code: statusCode });

      // Device unlinked from the phone — the stored session is dead and
      // every future connectSocket() would hit the same loggedOut status
      // forever. Clear the stale credentials and start a fresh session
      // so Baileys issues a new `qr` event instead of crash-looping.
      if (statusCode === DisconnectReason.loggedOut) {
        process.stderr.write("Logged out — clearing session and requesting re-pair.\n");
        try {
          sock?.ws?.close();
        } catch {
          // Best effort
        }
        clearAuthState();
        reconnectAttempts = 0;
        setTimeout(connectSocket, 1000);
        return;
      }

      // Transient — reconnect with backoff
      if (!shuttingDown && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        const backoff = Math.min(5000 * Math.pow(1.5, reconnectAttempts - 1), 60000);
        process.stderr.write(
          `Reconnecting in ${Math.round(backoff / 1000)}s (attempt ${reconnectAttempts})...\n`,
        );
        setTimeout(connectSocket, backoff);
      } else if (!shuttingDown) {
        process.stderr.write("Max reconnect attempts reached.\n");
        shutdown(1);
      }
    }
  });

  // --- Message events ---
  sock.ev.on("messages.upsert", ({ messages: msgs, type }) => {
    for (const msg of msgs) {
      const jid = msg.key?.remoteJid;
      if (!jid) continue;

      // Store the raw message
      storeMessage(jid, msg);

      // Emit generic message event
      const text = extractText(msg.message);
      emit({
        type: "message",
        jid,
        msg_id: msg.key?.id,
        from_me: Boolean(msg.key?.fromMe),
        text: text || null,
        timestamp: toEpoch(msg.messageTimestamp),
        push_name: msg.pushName || null,
        upsert_type: type,
      });

      // Download audio messages in background (non-blocking)
      if (isAudioMessage(msg.message)) {
        const audioMeta = getAudioMeta(msg.message);
        downloadAudio(msg).then((audioPath) => {
          if (audioPath) {
            emit({
              type: "audio_downloaded",
              jid,
              msg_id: msg.key?.id,
              from_me: Boolean(msg.key?.fromMe),
              audio_path: audioPath,
              duration_seconds: audioMeta?.seconds || 0,
              ptt: audioMeta?.ptt || false,
              timestamp: toEpoch(msg.messageTimestamp),
              push_name: msg.pushName || null,
            });
          }
        }).catch((err) => {
          process.stderr.write(`Audio download error: ${err.message}\n`);
        });
      }

      // Self-chat detection — save store immediately so Python can
      // read quoted context from store.json without waiting for the
      // 10s periodic save.
      if (isSelfChat(jid)) {
        saveStore();
        emitSelfChatMessage(msg);
      }
    }
  });

  // --- Outbound message delivery acks ---
  // WAMessageStatus: 1=PENDING, 2=SERVER_ACK, 3=DELIVERY_ACK, 4=READ.
  // We only treat >=3 as proof the recipient received it; server-ack is
  // not enough (message may still be queued at WhatsApp).
  sock.ev.on("messages.update", (updates) => {
    for (const u of updates) {
      const s = u.update?.status;
      if (typeof s === "number" && s >= 3 && u.key?.fromMe) {
        emit({
          type: "message_ack",
          msg_id: u.key.id,
          status: s,
          jid: u.key.remoteJid,
          ts: Date.now() / 1000,
        });
      }
    }
  });

  // --- Chat events ---
  sock.ev.on("chats.upsert", (chats) => {
    for (const chat of chats) {
      store.chats[chat.id] = { ...store.chats[chat.id], ...chat };
    }
  });

  sock.ev.on("chats.update", (updates) => {
    for (const update of updates) {
      if (store.chats[update.id]) {
        Object.assign(store.chats[update.id], update);
      }
    }
  });

  sock.ev.on("chats.delete", (ids) => {
    for (const id of ids) {
      delete store.chats[id];
    }
  });

  // --- Contact events ---
  sock.ev.on("contacts.upsert", (contacts) => {
    for (const contact of contacts) {
      store.contacts[contact.id] = { ...store.contacts[contact.id], ...contact };
    }
  });

  sock.ev.on("contacts.update", (updates) => {
    for (const update of updates) {
      if (store.contacts[update.id]) {
        Object.assign(store.contacts[update.id], update);
      }
    }
  });

  // --- History sync (first connection) ---
  sock.ev.on("messaging-history.set", ({ chats, contacts, messages }) => {
    if (chats) {
      for (const chat of chats) {
        store.chats[chat.id] = { ...store.chats[chat.id], ...chat };
      }
    }
    if (contacts) {
      for (const contact of contacts) {
        store.contacts[contact.id] = { ...store.contacts[contact.id], ...contact };
      }
    }
    if (messages) {
      for (const msg of messages) {
        const jid = msg.key?.remoteJid;
        if (jid) storeMessage(jid, msg);
      }
    }
    // Save immediately after history sync
    saveStore();
  });
}

// ---------------------------------------------------------------------------
// Periodic tasks
// ---------------------------------------------------------------------------

function startPeriodicTasks() {
  stopPeriodicTasks();

  // Save store every 10s
  storeInterval = setInterval(saveStore, STORE_SAVE_INTERVAL_MS);

  // Poll self-chat every 60s to catch messages that messages.upsert may miss
  selfChatPollInterval = setInterval(pollSelfChat, SELF_CHAT_POLL_INTERVAL_MS);
}

function stopPeriodicTasks() {
  if (storeInterval) {
    clearInterval(storeInterval);
    storeInterval = null;
  }
  if (selfChatPollInterval) {
    clearInterval(selfChatPollInterval);
    selfChatPollInterval = null;
  }
}

/**
 * Poll recent self-chat messages from the in-memory store and emit
 * any we haven't already emitted. This catches phone-originated
 * messages that messages.upsert may have missed.
 */
function pollSelfChat() {
  if (!myBareJid && !myBareLid) return;

  // Collect all JIDs that represent self-chat:
  // 1. Phone JID (@s.whatsapp.net)
  // 2. Linked-device JIDs (@lid) — self-chat from phone in multi-device mode
  // 3. Any other @lid JIDs that match our LID
  const candidates = new Set();

  if (myBareJid) {
    candidates.add(`${myBareJid}@s.whatsapp.net`);
  }
  if (myBareLid) {
    candidates.add(`${myBareLid}@lid`);
  }

  // Also scan all store keys for any @lid JIDs that match isSelfChat()
  for (const jid of Object.keys(store.messages)) {
    if (isSelfChat(jid)) {
      candidates.add(jid);
    }
  }

  for (const jid of candidates) {
    const msgs = store.messages[jid];
    if (!Array.isArray(msgs)) continue;
    for (const msg of msgs) {
      emitSelfChatMessage(msg);
    }
  }
}

// ---------------------------------------------------------------------------
// STDIN command handling
// ---------------------------------------------------------------------------

function setupStdinHandler() {
  const rl = readline.createInterface({ input: process.stdin, terminal: false });

  rl.on("line", async (line) => {
    let cmd;
    try {
      cmd = JSON.parse(line.trim());
    } catch {
      return; // Ignore malformed input
    }

    const id = cmd.id || null;

    try {
      switch (cmd.cmd) {
        case "send": {
          if (!sock || !sock.user) {
            emitResponse(id, false, "Not connected");
            break;
          }
          const to = String(cmd.to || "").trim();
          const text = String(cmd.text || "").trim();
          if (!to || !text) {
            emitResponse(id, false, "Missing 'to' or 'text'");
            break;
          }
          // Ensure JID format — strip leading "+" (Baileys JIDs are pure numbers)
          const cleanTo = to.replace(/^\+/, "");
          let jid = cleanTo.includes("@") ? cleanTo : `${cleanTo}@s.whatsapp.net`;

          // Resolve the canonical JID for non-@lid phone-style JIDs.
          // Baileys' onWhatsApp() is the only reliable way to:
          //   (a) check whether the contact is actually on WhatsApp
          //       (sock.sendMessage silently no-ops otherwise),
          //   (b) get the JID WhatsApp expects — Brazilian numbers
          //       registered before the 2010 mobile-9 prefix change
          //       are addressed without the 9 even when the contact
          //       saved the new format, so 5551996669496 must be
          //       sent to 555196669496@s.whatsapp.net to land.
          // @lid JIDs (self-chat, multi-device link IDs) are skipped:
          // they're already canonical.
          if (jid.endsWith("@s.whatsapp.net")) {
            try {
              const results = await sock.onWhatsApp(jid);
              const match = Array.isArray(results) && results[0];
              if (!match || !match.exists) {
                emitResponse(id, false, `Contact not on WhatsApp: ${cleanTo}`);
                break;
              }
              if (match.jid && match.jid !== jid) {
                jid = match.jid;
              }
            } catch (lookupErr) {
              // onWhatsApp failure shouldn't block legitimate sends —
              // fall through with the original JID. Baileys may still
              // succeed via local cache; if it can't reach the
              // contact, the delivery ack will never arrive.
              emit({
                type: "log",
                level: "warn",
                message: `onWhatsApp lookup failed for ${jid}: ${lookupErr.message || lookupErr}`,
              });
            }
          }

          // Optional caller-supplied messageId enables idempotent re-sends
          // (WhatsApp server dedupes by ID per sender), used by the listener
          // outbox to re-process leftover requests safely on restart.
          const sendOpts = cmd.messageId ? { messageId: String(cmd.messageId) } : undefined;
          const sendTimeout = 30_000;
          const result = await Promise.race([
            sock.sendMessage(jid, { text }, sendOpts),
            new Promise((_, reject) =>
              setTimeout(() => reject(new Error(`sendMessage timed out after ${sendTimeout / 1000}s`)), sendTimeout),
            ),
          ]);
          emitResponse(id, true, {
            message_id: result?.key?.id || null,
            resolved_jid: jid,
          });
          break;
        }

        case "status": {
          const connected = Boolean(sock?.user);
          emitResponse(id, true, {
            connected,
            jid: myBareJid || null,
            lid: myBareLid || null,
            user: sock?.user || null,
          });
          break;
        }

        default:
          emitResponse(id, false, `Unknown command: ${cmd.cmd}`);
      }
    } catch (err) {
      emitResponse(id, false, err.message || String(err));
    }
  });

  rl.on("close", () => {
    // Parent process closed stdin — shut down gracefully
    shutdown(0);
  });
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;

  stopPeriodicTasks();

  // Final store save
  try {
    saveStore();
  } catch {
    // Best effort
  }

  // Close the socket
  try {
    sock?.ws?.close();
    sock?.end?.();
  } catch {
    // Best effort
  }

  setTimeout(() => process.exit(code), 500);
}

process.on("SIGTERM", () => shutdown(0));
process.on("SIGINT", () => shutdown(0));
process.on("uncaughtException", (err) => {
  process.stderr.write(`Uncaught exception: ${err.message}\n${err.stack}\n`);
  shutdown(1);
});
process.on("unhandledRejection", (err) => {
  process.stderr.write(`Unhandled rejection: ${err}\n`);
  // Don't exit — Baileys can generate spurious rejections
});

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  process.stderr.write(`Arandu WhatsApp client starting (auth: ${AUTH_DIR})\n`);

  loadStore();
  setupStdinHandler();
  await connectSocket();
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message}\n${err.stack}\n`);
  process.exit(1);
});
