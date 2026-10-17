import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// The Python harness (../server.py) owns /api. In `next dev` the page proxies
// to it; `next build` exports static files that server.py serves itself.
const API = process.env.ONEPOLICY_API ?? "http://127.0.0.1:8000";

export default function config(phase: string): NextConfig {
  const base: NextConfig = {
    turbopack: {
      rules: {
        "*.css": { loaders: ["@tailwindcss/turbopack"], as: "*.css" },
      },
    },
  };
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      ...base,
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
      },
    };
  }
  // trailingSlash: /train -> out/train/index.html, which server.py looks up.
  return { ...base, output: "export", trailingSlash: true, images: { unoptimized: true } };
}
