import { courseInfo, schedule, removedContent } from "@/data/nlp";
import ScheduleTable, { toHtml } from "@/components/ScheduleTable";

// Unlisted instructor view. Not in the course nav and not linked from any page.
// `robots: noindex, nofollow` keeps it out of search results — but note this is
// obscurity, not access control: anyone who knows or guesses the URL can read it
// once the course site is published.
export const metadata = {
  title: "Schedule Planning – Natural Language Processing",
  robots: { index: false, follow: false },
};

export default function NLPPlanningPage() {
  const hiddenCount = schedule.filter((r) => r.materialsHidden && r.materials).length;
  const withMaterials = schedule.filter((r) => r.materials).length;

  return (
    <div>
      <div className="mb-8 rounded-lg border border-dashed border-amber-400 bg-amber-50 p-5">
        <h1 className="text-lg font-bold mb-1">Schedule Planning — Instructor View</h1>
        <p className="text-sm leading-relaxed">
          The full schedule, including materials hidden from the public page at{" "}
          <a href="/nlp" className="text-accent underline hover:text-accent/80">
            /nlp
          </a>
          . This page is unlisted: it is not in the course navigation and is not
          linked from anywhere, but it is <strong>not private</strong> — anyone with
          the URL can read it once the course site is published.
        </p>
        <p className="text-sm mt-3 leading-relaxed">
          To hide a row&rsquo;s readings from students, set{" "}
          <code className="bg-white px-1 py-0.5 rounded text-xs">
            materialsHidden: true
          </code>{" "}
          on that entry in <code className="bg-white px-1 py-0.5 rounded text-xs">data/nlp.ts</code>.
          Hidden rows stay visible here, tagged in amber.
        </p>
      </div>

      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-1">
          {courseInfo.number}: {courseInfo.title}
        </h2>
        <p className="text-text-light">
          {courseInfo.institution} &middot; {courseInfo.semester} &middot;{" "}
          {courseInfo.time}
        </p>
      </div>

      <p className="text-sm text-text-light mb-4">
        {schedule.length} sessions &middot; {withMaterials} with materials &middot;{" "}
        {hiddenCount} hidden from students
      </p>

      <ScheduleTable rows={schedule} revealHidden />

      {/* Parked topics. Lives here rather than on /nlp so it can never reach
          students, even if entries are added later. */}
      {removedContent.length > 0 && (
        <div className="mb-10 rounded-lg border border-dashed border-gray-400 bg-gray-50 p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-text-light mb-1">
            Removed Content
          </h2>
          <p className="text-xs text-text-light mb-4 italic">
            Parked topics — kept so their readings are not lost.
          </p>
          <ul className="space-y-4">
            {removedContent.map((item) => (
              <li key={item.topic}>
                <p className="font-medium text-sm">{item.topic}</p>
                {item.note && (
                  <p className="text-xs text-text-light italic">{item.note}</p>
                )}
                {item.materials && (
                  <div className="text-sm mt-1" dangerouslySetInnerHTML={{ __html: toHtml(item.materials) }} />
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
