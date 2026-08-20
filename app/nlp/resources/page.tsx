import { resources } from "@/data/nlp";

export const metadata = {
  title: "Resources – Natural Language Processing",
};

export default function ResourcesPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Resources</h1>
      <ul className="space-y-4">
        {resources.map((r) => (
          <li key={r.url}>
            <a
              href={r.url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-accent underline hover:text-accent/80"
            >
              {r.title}
            </a>
            {r.description && (
              <span className="text-sm text-text-light"> — {r.description}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
