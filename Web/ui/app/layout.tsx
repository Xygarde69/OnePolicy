import type { Metadata } from "next";
import { Archivo_Black, JetBrains_Mono, Pacifico, Space_Grotesk } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SiteHeader } from "@/components/brutal/site-header";
import "./globals.css";

const display = Archivo_Black({ variable: "--font-display", weight: "400", subsets: ["latin"] });
const body = Space_Grotesk({ variable: "--font-body", subsets: ["latin"] });
const code = JetBrains_Mono({ variable: "--font-code", subsets: ["latin"] });
// The OnePolicy™ wordmark (components/brutal/brand.tsx).
const brand = Pacifico({ variable: "--font-brand", weight: "400", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "OnePolicy™",
  description: "Train your own quadruped, or walk any quadruped with OnePolicy™.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${code.variable} ${brand.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <TooltipProvider>
          <SiteHeader />
          <main className="flex flex-1 flex-col">{children}</main>
          <Toaster position="bottom-right" />
        </TooltipProvider>
      </body>
    </html>
  );
}
