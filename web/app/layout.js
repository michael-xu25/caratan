import "./globals.css";

export const metadata = {
  title: "Caratan — teaching a small LLM to play Catan with RL",
  description:
    "Caratan: teaching Qwen3-8B to play 1v1 Settlers of Catan via GRPO reinforcement learning. Held-out results, training curves, and interactive game replays.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
