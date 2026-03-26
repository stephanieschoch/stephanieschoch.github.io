export const metadata = {
  title: "CV – Stephanie Schoch",
};

export default function CVPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Curriculum Vitae</h1>
      <p className="text-text-light text-sm mb-6">
        <a
          href="/cv.pdf"
          download
          className="text-accent hover:text-accent-dark underline"
        >
          Download PDF
        </a>
      </p>
      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <iframe
          src="/cv.pdf"
          className="w-full border-0"
          style={{ height: "80vh" }}
          title="Curriculum Vitae"
        />
      </div>
    </div>
  );
}
