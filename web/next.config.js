/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  outputFileTracingRoot: __dirname,
};

module.exports = nextConfig;
