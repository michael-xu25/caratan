import { Space_Grotesk } from "next/font/google";
import "./globals.css";

// Display font with character for headings — replaces the system default.
// Self-hosted at build time (works with static export, no runtime dependency).
const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

export const metadata = {
  title: "Caratan — teaching a small LLM to play Catan with RL",
  description:
    "Caratan: teaching Qwen3-8B to play 1v1 Settlers of Catan via GRPO reinforcement learning. Held-out results, training curves, and interactive game replays.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={display.variable}>
      <body>{children}</body>
    </html>
  );
}
