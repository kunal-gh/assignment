import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Resume Screener - Intelligent Candidate Ranking",
  description: "AI-powered resume screening with semantic analysis and fairness detection",
  keywords: ["AI", "resume", "screening", "recruitment", "ML", "NLP"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.Node;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
