import CourseHeader from "@/components/CourseHeader";
import Footer from "@/components/Footer";

const navLinks = [
  { href: "/nlp", label: "Home" },
  { href: "/nlp#schedule", label: "Schedule" },
  { href: "/nlp/resources", label: "Resources" },
  { href: "/nlp/project", label: "Project" },
];

export default function NLPLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <CourseHeader
        courseName="Natural Language Processing"
        basePath="/nlp"
        navLinks={navLinks}
      />
      <main className="max-w-4xl mx-auto px-6 py-10 flex-1 w-full">
        {children}
      </main>
      <Footer />
    </>
  );
}
