import type { ScheduleRow } from "@/data/nlp";

// Each line becomes its own block: a "Readings:" / "Suggested Papers:" label
// followed inline by its comma-separated links.
export function toHtml(markdown: string) {
  return markdown
    .split("\n")
    .filter((line) => line.trim())
    .map((line) =>
      line
        // [text](url) or [text](url "hover title")
        .replace(
          /\[([^\]]+)\]\(([^\s)]+)(?:\s+"([^"]*)")?\)/g,
          (_m, text: string, url: string, title?: string) =>
            `<a href="${url}" target="_blank" rel="noopener noreferrer"` +
            `${title ? ` title="${title}"` : ""}` +
            ` class="text-accent underline hover:text-accent/80">${text}</a>`
        )
    )
    .map((line) => `<div class="mb-1 last:mb-0">${line}</div>`)
    .join("");
}

interface ScheduleTableProps {
  rows: ScheduleRow[];
  /**
   * Show materials even on rows flagged `materialsHidden`, tagged so it is
   * obvious they are not public. Used by the instructor view at /nlp/planning.
   */
  revealHidden?: boolean;
}

export default function ScheduleTable({ rows, revealHidden = false }: ScheduleTableProps) {
  return (
    <div className="overflow-x-auto mb-10 rounded-lg border border-gray-300">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-[#f0f0f0] text-left">
            <th className="py-3 px-4 font-semibold w-16">Week</th>
            <th className="py-3 px-4 font-semibold w-32">Date</th>
            <th className="py-3 px-4 font-semibold w-1/3">Topic</th>
            <th className="py-3 px-4 font-semibold">Course Material</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const hidden = Boolean(row.materialsHidden);
            const show = row.materials && (revealHidden || !hidden);
            return (
              <tr key={i} className="bg-white border-b border-gray-200">
                <td className="py-3 px-4 text-text-light">{row.week}</td>
                <td className="py-3 px-4 text-text-light">{row.date}</td>
                <td className="py-3 px-4">
                  {revealHidden && row.planningTopic ? row.planningTopic : row.topic}
                  {revealHidden && row.planningTopic && row.planningTopic !== row.topic && (
                    <span className="block text-xs text-text-light italic mt-0.5">
                      students see: {row.topic}
                    </span>
                  )}
                  {row.due && (
                    <span className="ml-2 text-xs font-medium text-accent bg-accent/10 px-2 py-0.5 rounded">
                      {row.due}
                    </span>
                  )}
                </td>
                <td className="py-3 px-4">
                  {show && (
                    <>
                      {revealHidden && hidden && (
                        <span className="inline-block mb-1 text-xs font-medium text-amber-800 bg-amber-100 px-2 py-0.5 rounded">
                          hidden from students
                        </span>
                      )}
                      <div
                        className={`text-sm ${revealHidden && hidden ? "opacity-70" : ""}`}
                        dangerouslySetInnerHTML={{ __html: toHtml(row.materials!) }}
                      />
                    </>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
