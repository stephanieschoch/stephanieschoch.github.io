import { publications } from "@/data/publications";

export const metadata = {
  title: "Publications – Stephanie Schoch",
};

export default function PublicationsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Publications</h1>
      <div className="space-y-4">
        {publications.map((pub, i) => (
          <div key={i} className="bg-white rounded-lg p-5 shadow-sm">
            <h3 className="font-semibold text-lg">
              {pub.link ? (
                <a href={pub.link} target="_blank" rel="noopener noreferrer" className="text-text hover:text-accent">
                  {pub.title}
                </a>
              ) : (
                pub.title
              )}
            </h3>
            <p className="text-text-light text-sm mt-1">{pub.authors}</p>
            <p className="text-text-light text-sm">
              <span className="italic">{pub.venue}</span>, {pub.year}
            </p>
            {pub.note && (
              <p className="text-accent text-sm font-medium mt-1">{pub.note}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
