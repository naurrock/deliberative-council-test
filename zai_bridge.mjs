#!/usr/bin/env bun
/**
 * Bridge script: accepts a JSON request on stdin, calls z-ai SDK, returns JSON on stdout.
 * This avoids the overhead of CLI argument parsing and subprocess spawning.
 *
 * Input format:
 * {
 *   "model": "optional-model-name",
 *   "system": "optional system prompt",
 *   "user": "user prompt (required)",
 *   "max_tokens": 4096,
 *   "temperature": 0.7
 * }
 *
 * Output format (on stdout):
 * {
 *   "ok": true,
 *   "content": "response text",
 *   "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N }
 * }
 * OR
 * { "ok": false, "error": "error message" }
 */

import ZAI from "/home/z/.bun/install/global/node_modules/z-ai-web-dev-sdk/dist/index.js";

async function main() {
  let input;
  try {
    const chunks = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk);
    }
    input = JSON.parse(Buffer.concat(chunks).toString());
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: `Failed to parse stdin: ${e.message}` }));
    process.exit(0);
  }

  if (!input.user) {
    console.log(JSON.stringify({ ok: false, error: "Missing 'user' field" }));
    process.exit(0);
  }

  try {
    const zai = await ZAI.create();

    const messages = [];
    if (input.system) {
      messages.push({ role: "system", content: input.system });
    }
    messages.push({ role: "user", content: input.user });

    const completion = await zai.chat.completions.create({
      messages,
      temperature: input.temperature ?? 0.7,
      max_tokens: input.max_tokens ?? 4096,
    });

    const content = completion.choices[0]?.message?.content || "";
    const usage = completion.usage || {};

    console.log(JSON.stringify({
      ok: true,
      content,
      usage: {
        prompt_tokens: usage.prompt_tokens || 0,
        completion_tokens: usage.completion_tokens || 0,
        total_tokens: usage.total_tokens || 0,
      }
    }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: e.message }));
    process.exit(0);
  }
}

main();
