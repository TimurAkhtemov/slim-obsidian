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

test("a stream stopped before its request was made rejects at once, with no request", async () => {
  let requests = 0;
  const server = http.createServer((_req, res) => { requests += 1; res.end(); });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const controller = new AbortController();
  controller.abort();
  try {
    await assert.rejects(streamRequest(server.address().port, "/api/copilot/stream", {}, {}, controller.signal),
                         { name: "AbortError" });
    assert.equal(requests, 0);
  } finally {
    server.close();
  }
});
