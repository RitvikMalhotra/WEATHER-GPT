/**
 * Local launcher for WeatherGPT.
 *
 * There is one process to start, not two. The browser client is not a separate
 * dev server: `app/voice/router.py` serves page.html, styles.css and app.js from
 * the same FastAPI application that serves /api/v1, so the UI and the API share
 * an origin and the page's relative fetches reach the API with no proxy and no
 * CORS configuration. Starting a second server would give the page a different
 * origin and break exactly the thing it is supposed to make easier.
 *
 * So this file adds no orchestration. It spawns the documented uvicorn command,
 * prints where each surface is, and gets out of the way — no dependencies, and
 * nothing here that a person could not type themselves.
 *
 * Two details it does own, both of which are easy to get wrong by hand:
 *
 *   - The working directory is `backend/`, because pydantic-settings reads
 *     `.env` relative to the process's cwd. Launched from the repository root,
 *     every configured value silently falls back to its default.
 *   - The port is 8000 unless PORT says otherwise, and AI_BACKEND_BASE_URL is
 *     derived from it. The AI layer reaches its own service over HTTP, so a
 *     port change that missed that variable would leave the REST endpoints
 *     working while every conversational answer came back unavailable.
 */

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const backend = join(root, "backend");

const host = process.env.HOST ?? "127.0.0.1";
const port = process.env.PORT ?? "8000";
const python = process.env.PYTHON ?? "python";
const reload = process.argv.includes("--reload");

const args = [
  "-m",
  "uvicorn",
  "app.main:app",
  "--host",
  host,
  "--port",
  port,
  ...(reload ? ["--reload"] : []),
];

const url = `http://${host === "0.0.0.0" ? "127.0.0.1" : host}:${port}`;

console.log("");
console.log("  WeatherGPT");
console.log("  ─────────────────────────────────────────────");
console.log(`  FRONTEND   ${url}/            (also ${url}/voice)`);
console.log(`  BACKEND    ${url}/api/v1      (docs at ${url}/docs)`);
console.log("");
console.log("  One FastAPI process serves both. Ctrl+C stops it.");
if (reload) console.log("  Reload is on: the server restarts when a file changes.");
console.log("");

const child = spawn(python, args, {
  cwd: backend,
  stdio: "inherit",
  // Inherited so backend/.env, and anything already exported, keep deciding
  // configuration. Only the AI layer's self-address is filled in, and only
  // when it has not already been set.
  env: {
    ...process.env,
    AI_BACKEND_BASE_URL:
      process.env.AI_BACKEND_BASE_URL ?? `http://127.0.0.1:${port}`,
  },
});

child.on("error", (error) => {
  if (error.code === "ENOENT") {
    console.error(
      `\n  Could not run '${python}'. Install Python 3.12+, or set PYTHON to its path.\n`
    );
  } else {
    console.error(`\n  Could not start the backend: ${error.message}\n`);
  }
  process.exit(1);
});

// Ctrl+C reaches the child on its own — it is in this console's process group —
// so this handler exists only to stop Node from exiting first and orphaning it.
// The child's own exit is what ends this process, with its status.
process.on("SIGINT", () => {});
process.on("SIGTERM", () => {
  child.kill("SIGTERM");
});

child.on("exit", (code, signal) => {
  process.exit(signal ? 0 : code ?? 0);
});
