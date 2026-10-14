// The copilot's event stream.
import assert from "node:assert/strict";
import test from "node:test";
import http from "node:http";
import { createRequire } from "node:module";
import "./copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { SSEParser, streamRequest } = require("../src/sse.js");

test("SSE parsing survives arbitrary chunk boundaries", () => {
  const parser = new SSEParser();
  assert.deepEqual(parser.push("event: sta"), []);
  assert.deepEqual(parser.push("ge\ndata: {\"stage\":\"Writing\"}\n\nevent: delta\n"), [
    { event: "stage", data: { stage: "Writing" } },
  ]);
  assert.deepEqual(parser.push("data: {\"text\":\"Hi\"}\n\n"), [
    { event: "delta", data: { text: "Hi" } },
  ]);
});

test("an SSE error handler rejects streamRequest instead of escaping the callback", async () => {
  const server = http.createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "text/event-stream" });
    response.end('event: error\ndata: {"error":"offline"}\n\n');
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    await assert.rejects(
      streamRequest(server.address().port, "/copilot", {}, {
        error(data) { throw new Error(data.error); },
      }),
      /offline/,
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
