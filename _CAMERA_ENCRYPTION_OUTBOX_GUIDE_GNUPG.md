# 🎥 Camera-Side Encryption & OUTBOX Guide (GnuPG Edition)
### Capture • Encrypt • Queue • Ship (Safely)

## 🔗 This guide pairs with

- **CAMERA_ENCRYPTION_OUTBOX_GUIDE.md** or **CAMERA_ENCRYPTION_OUTBOX_GUIDE_GNUPG.md** — camera-side encryption & OUTBOX
- **SECURE_CAMERA_RECEIVER_SYNC.md** — SSH transport & automation
- **RECEIVER_VERIFICATION_INGEST_GUIDE.md** — receiver-side verification & ingest

These guides form a single, coherent pipeline and are intended to be used together.


---

## 🧠 Purpose

This document defines the **camera-side responsibilities** in the secure capture pipeline, using **GnuPG (gpg)** as the encryption tool.

The camera node is the **only place where plaintext exists**.  
Its job is to:
- Capture video reliably
- Encrypt immediately after finalization
- Queue encrypted artifacts
- Never export plaintext

This guide replaces the `age` variant with a **GnuPG-based default** suitable for Raspberry Pi OS.

---

## 🎯 Camera Goals

🟢 Plaintext never leaves the node  
🟢 Encryption is automatic and deterministic  
🟢 OUTBOX behaves like a safe queue  
🟢 Failures do not cause data loss  
🟢 Artifacts are verifiable downstream  

---

## 🧱 Trust Boundary (Critical)

```
┌─────────────────────────────┐
│ CAMERA NODE (TRUSTED)       │
│ ├─ Plaintext allowed        │
│ ├─ LUKS at rest             │
│ └─ Encryption authority     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ OUTBOX (UNTRUSTED ZONE)     │
│ ├─ Encrypted only (.gpg)    │
│ ├─ Safe to copy/ship        │
│ └─ No secrets inside        │
└─────────────────────────────┘
```

Once data enters **OUTBOX**, it must be assumed hostile environments await.

---

## 📁 Canonical Directory Layout

```
/media/user/disk/
├── videos/        # plaintext clips (local only)
├── outbox/        # encrypted artifacts (export only)
├── logs/          # camera + detection logs
└── tmp/           # transient working files
```

❌ OUTBOX must never contain plaintext  
❌ videos/ must never be synced  

---

## 🔐 Encryption Strategy (GnuPG)

### Why GnuPG on Raspberry Pi OS

- Installed or one-command install
- Mature, audited, widely understood
- Supports asymmetric encryption
- No private keys required on camera

⚠️ Complexity is higher than `age`, so **strict conventions are mandatory**.

---

## 🔑 Key Setup (Once)

### On receiver or secure admin system
Generate keypair:
```bash
gpg --full-generate-key
```

Export **public key only**:
```bash
gpg --armor --export receiver@example.com > receiver.pub.asc
```

### On camera
Import public key:
```bash
gpg --import receiver.pub.asc
```

Lock permissions:
```bash
chmod 700 ~/.gnupg
chmod 600 ~/.gnupg/*
```

🚨 **Private keys must never be present on the camera.**

---

## 🎬 Clip Finalization Flow

1. Camera records clip to `videos/`
2. Clip is closed and stable
3. Encryption job triggers
4. Encrypted file written to `outbox/`
5. Hash + manifest updated
6. Plaintext handled per retention policy

---

## 🔁 Encrypt + Queue (Canonical Command)

```bash
gpg --batch --yes --trust-model always   --recipient receiver@example.com   --output /media/user/disk/outbox/clip_20260115_1200.mp4.gpg   --encrypt /media/user/disk/videos/clip_20260115_1200.mp4
```

### Notes
- `--batch` ensures non-interactive operation
- `--trust-model always` avoids trustdb prompts
- Recipient must match imported public key UID

---

## 🔐 Generate Integrity Hash

```bash
sha256sum /media/user/disk/outbox/clip_20260115_1200.mp4.gpg   > /media/user/disk/outbox/clip_20260115_1200.mp4.gpg.sha256
```

---

## 📜 Manifest Design (Strongly Recommended)

Example `manifest.json` entry:
```json
{
  "clip": "clip_20260115_1200.mp4.gpg",
  "sha256": "abc123...",
  "timestamp": "2026-01-15T12:00:00Z",
  "camera_id": "cam1",
  "software": "AI_CAM_VIDSec v1.0",
  "encryption": "gpg"
}
```

---

## 🧹 Plaintext Retention Policy

Choose **one**:

### Option A — Short Retention (recommended)
- Keep plaintext for N hours
- Auto-delete after verification

### Option B — No Retention
- Delete plaintext immediately after encryption

### Option C — Manual Review Window
- Retain until operator confirmation

🚨 Plaintext deletion must be deliberate and logged.

---

## 🧪 Validation Before Shipping

Before OUTBOX sync:
```bash
ls -lh outbox/
sha256sum -c *.sha256
```

Only verified files may leave the node.

---

## 🧰 Automation Options

- Post-recording hook
- systemd path unit watching `videos/`
- Periodic scan for unencrypted clips

**Rule:** encryption must be idempotent.

---

## 📴 Offline Safety

OUTBOX is safe to:
- Copy to USB
- Transmit over hostile networks
- Store on staging disks

Because:
- No plaintext
- No private keys
- Integrity verifiable

---

## ❌ Camera Anti-Patterns

🚫 Storing private keys on camera  
🚫 Encrypting during recording  
🚫 Syncing `videos/`  
🚫 Re-encrypting already encrypted files  
🚫 Trusting filenames alone  

---

## 🧠 Operational Philosophy

> **The camera is the root of trust.**

If encryption fails → stop exporting  
If verification fails → stop syncing  

---

## ✅ Camera Checklist

- [ ] LUKS enabled
- [ ] GPG public key only
- [ ] OUTBOX encrypted-only
- [ ] Hashes generated
- [ ] Manifests updated
- [ ] Plaintext retention defined

---

**End of document**


## 📁 Standard Directory Layout (Project-Wide)

Unless explicitly stated otherwise, all guides use the following layout for **encrypted ingest data**:

```
<BASE_PATH>/
├── encrypted/     # encrypted artifacts (.gpg / .age)
├── hashes/        # integrity hashes (.sha256)
├── manifests/     # JSON manifests
└── quarantine/    # failed or unverified files
```

Notes:
- Plaintext is **never** stored here
- Only `encrypted/` is transported between systems
- `quarantine/` is for investigation only and is never synced
