# OnePolicy web UI

Next.js (static export) + TypeScript + Tailwind CSS 4 + shadcn/ui (radix),
three.js via @react-three/fiber, motion, recharts. Neo-brutalist theme in
`app/globals.css` (tokens on `:root`, `.brutal` surfaces).

```bash
npm install
npm run build        # -> out/, served by ../server.py at http://localhost:8000
npm run dev          # http://localhost:3000, /api proxied to server.py on :8000
```

- `app/`: landing, `/train` + `/train/quadruped` (upload → check → rewards →
  limits → conditions → train → policy), `/onepolicy` + `/onepolicy/quadruped`
  (preloaded / generate / upload → parts → terrain → WASD in the sim).
- `components/flow/`: the steps; `components/three/`: robot rendering from
  `/api/geometry` (`../geometry.py`); `components/brutal/`: shared pieces.
- `lib/api.ts`: typed client for every `server.py` endpoint.
