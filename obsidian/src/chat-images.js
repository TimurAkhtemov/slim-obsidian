// Images pasted into the copilot: checked, named and resized.
"use strict";

const MAX_COPILOT_IMAGES = 3;
const MAX_COPILOT_IMAGE_BYTES = 2_000_000;
const MAX_COPILOT_IMAGE_INPUT_BYTES = 20_000_000;
const MAX_COPILOT_IMAGE_EDGE = 1600;
const COPILOT_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function imageBytes(data) {
  return Buffer.byteLength(String(data || ""), "base64");
}

function imageName(name, mime) {
  const safe = String(name || "image").replace(/\\/g, "/").split("/").pop().slice(0, 120);
  const ext = mime === "image/png" ? "png" : mime === "image/webp" ? "webp" : "jpg";
  return `${safe.replace(/\.[^.]+$/, "") || "image"}.${ext}`;
}

function loadBrowserImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("This image could not be decoded."));
    image.src = source;
  });
}

async function resizeCopilotImage(file) {
  const mime = String(file?.type || "").toLowerCase();
  if (!COPILOT_IMAGE_TYPES.has(mime)) throw new Error("Use a PNG, JPEG, or WebP image.");
  if (!Number.isFinite(file.size) || file.size <= 0) throw new Error("This image is empty.");
  if (file.size > MAX_COPILOT_IMAGE_INPUT_BYTES) throw new Error("Images must be 20 MB or smaller before resizing.");
  const buffer = Buffer.from(await file.arrayBuffer());
  const original = `data:${mime};base64,${buffer.toString("base64")}`;
  const image = await loadBrowserImage(original);
  if (Math.max(image.naturalWidth, image.naturalHeight) <= MAX_COPILOT_IMAGE_EDGE
      && buffer.length <= MAX_COPILOT_IMAGE_BYTES) {
    return { name: imageName(file.name, mime), mime, data: original.split(",")[1],
             preview: original, bytes: buffer.length };
  }
  let scale = Math.min(1, MAX_COPILOT_IMAGE_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    context.fillStyle = "#fff";              // transparent textbook captures stay legible
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const quality = Math.max(0.65, 0.9 - attempt * 0.07);
    const resized = canvas.toDataURL("image/jpeg", quality);
    const data = resized.split(",")[1];
    const bytes = imageBytes(data);
    if (bytes <= MAX_COPILOT_IMAGE_BYTES) {
      return { name: imageName(file.name, "image/jpeg"), mime: "image/jpeg", data,
               preview: resized, bytes };
    }
    scale *= 0.8;
  }
  throw new Error("The image is still over 2 MB after resizing. Crop it and try again.");
}

module.exports = { MAX_COPILOT_IMAGES, MAX_COPILOT_IMAGE_BYTES, MAX_COPILOT_IMAGE_INPUT_BYTES,
        MAX_COPILOT_IMAGE_EDGE, COPILOT_IMAGE_TYPES, imageBytes, imageName, loadBrowserImage,
        resizeCopilotImage };
