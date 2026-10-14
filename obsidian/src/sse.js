// Server-sent events from the copilot's stream.
"use strict";

const http = require("http");

class SSEParser {
  constructor() { this.buffer = ""; }
  push(chunk) {
    this.buffer += chunk;
    const events = [];
    while (true) {
      const boundary = this.buffer.indexOf("\n\n");
      if (boundary < 0) break;
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      try { events.push({ event, data: JSON.parse(data.join("\n")) }); }
      catch (_error) { events.push({ event: "error", data: { error: "Malformed server event" } }); }
    }
    return events;
  }
}

function streamRequest(port, path, body, handlers, signal) {
  return new Promise((resolve, reject) => {
    const encoded = Buffer.from(JSON.stringify(body));
    const request = http.request({
      hostname: "127.0.0.1", port, path, method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": encoded.length },
    }, (response) => {
      if (response.statusCode !== 200) {
        const parts = [];
        response.on("data", (chunk) => parts.push(chunk));
        response.on("end", () => reject(new Error(
          `${path} returned ${response.statusCode}: ${Buffer.concat(parts).toString("utf8")}`)));
        return;
      }
      const parser = new SSEParser();
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        try {
          for (const item of parser.push(chunk)) {
            const handler = handlers[item.event];
            if (handler) handler(item.data);
          }
        } catch (error) {
          response.destroy();
          request.destroy();
          reject(error);
        }
      });
      response.on("end", resolve);
      response.on("error", reject);
    });
    request.on("error", reject);
    if (signal) signal.addEventListener("abort", () => {
      const error = new Error("Cancelled");
      error.name = "AbortError";
      request.destroy(error);
    }, { once: true });
    request.end(encoded);
  });
}

module.exports = { SSEParser, streamRequest };
