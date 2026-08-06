import Header from "@/components/Header";
import Footer from "@/components/Footer";

export default function MainLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <Header />
      <main className="max-w-4xl mx-auto px-6 py-10 flex-1 w-full">
        {children}
      </main>
      <Footer />
    </>
  );
}
