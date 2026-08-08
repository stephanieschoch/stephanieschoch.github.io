import Image from "next/image";
import { courseInfo, schedule, deadlines } from "@/data/nlp";

export const metadata = {
  title: "Natural Language Processing – Stephanie Schoch",
};

export default function NLPPage() {
  return (
    <div>
      {/* Course Title */}
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold mb-1">
          {courseInfo.number}: {courseInfo.title}
        </h1>
        <p className="text-text-light">
          {courseInfo.institution} &middot; {courseInfo.semester}
        </p>
      </div>

      {/* Instructors */}
      <h3 className="font-semibold mb-3">Instructors</h3>
      <div className="flex flex-wrap gap-6 mb-10">
        {courseInfo.instructors.map((inst, i) => (
          <div key={i} className="flex flex-col items-center">
            <Image
              src="/headshot.jpg"
              alt={inst.name}
              width={80}
              height={80}
              className="w-20 h-20 rounded-full object-cover mb-2"
            />
            {inst.url ? (
              <a href={inst.url} className="text-sm text-text hover:text-accent font-medium">
                {inst.name}
              </a>
            ) : (
              <span className="text-sm font-medium">{inst.name}</span>
            )}
          </div>
        ))}
      </div>

      {/* Welcome Box */}
      <div className="bg-[#eef4fb] rounded-lg p-6 mb-10">
        <h3 className="font-semibold mb-2">Welcome!</h3>
        <p className="leading-relaxed">{courseInfo.description}</p>
      </div>

      {/* Schedule */}
      <h2 id="schedule" className="text-xl font-semibold mb-4 scroll-mt-20">Schedule</h2>
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
            {schedule.map((row, i) => (
              <tr key={i} className="bg-white border-b border-gray-200">
                <td className="py-3 px-4 text-text-light">{row.week}</td>
                <td className="py-3 px-4 text-text-light">{row.date}</td>
                <td className="py-3 px-4">
                  {row.topic}
                  {row.due && (
                    <span className="ml-2 text-xs font-medium text-accent bg-accent/10 px-2 py-0.5 rounded">
                      {row.due}
                    </span>
                  )}
                </td>
                <td className="py-3 px-4">
                  {row.materials && (
                    <div className="text-sm" dangerouslySetInnerHTML={{
                      __html: row.materials
                        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-accent underline hover:text-accent/80">$1</a>')
                        .replace(/\n(- .+(?:\n- .+)*)/g, (_match, items) => {
                          const lis = (items as string).split('\n').map((item: string) => `<li>${item.replace(/^- /, '')}</li>`).join('');
                          return `<ul class="list-disc ml-8 mt-1">${lis}</ul>`;
                        }),
                    }} />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Deadlines */}
      <h2 id="deadlines" className="text-xl font-semibold mb-4 scroll-mt-20">Deadlines</h2>
      <div className="overflow-x-auto mb-10 rounded-lg border border-gray-300">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-[#f0f0f0] text-left">
              <th className="py-3 px-4 font-semibold w-16">Week</th>
              <th className="py-3 px-4 font-semibold">Deadline</th>
              <th className="py-3 px-4 font-semibold w-40">Date</th>
              <th className="py-3 px-4 font-semibold w-36">Time</th>
            </tr>
          </thead>
          <tbody>
            {deadlines.map((row, i) => (
              <tr key={i} className="bg-white border-b border-gray-200">
                <td className="py-3 px-4 text-text-light">{row.week}</td>
                <td className="py-3 px-4">{row.deadline}</td>
                <td className="py-3 px-4 text-text-light">{row.date}</td>
                <td className="py-3 px-4 text-text-light">{row.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Overview */}
      <div className="bg-[#eef4fb] rounded-lg p-6 space-y-5">
        <h2 className="text-xl font-semibold">Overview</h2>
        <div>
          <h3 className="font-semibold mb-2">Course Info</h3>
          <ul className="text-sm space-y-1">
            <li><span className="font-medium">Time:</span> {courseInfo.time}</li>
            <li><span className="font-medium">Location:</span> {courseInfo.location}</li>
            <li><span className="font-medium">Office Hours:</span> {courseInfo.officeHours}</li>
          </ul>
        </div>
        <div>
          <h3 className="font-semibold mb-2">Prerequisites</h3>
          <p className="text-sm">{courseInfo.prerequisites}</p>
        </div>
      </div>
    </div>
  );
}
