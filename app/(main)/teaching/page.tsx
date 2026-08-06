import Link from "next/link";
import { courses, tutorials } from "@/data/teaching";

export const metadata = {
  title: "Teaching – Stephanie Schoch",
};

export default function TeachingPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Teaching</h1>

      <h2 className="text-xl font-semibold mb-4">Courses</h2>
      <div className="space-y-4 mb-10">
        {courses.map((course, i) => (
          <div key={i} className="bg-white rounded-lg p-5 shadow-sm">
            <h3 className="font-semibold text-lg">
              {course.link ? (
                <Link href={course.link} className="text-text hover:text-accent">
                  {course.name}
                </Link>
              ) : (
                course.name
              )}
            </h3>
            <p className="text-text-light text-sm mt-1">
              {course.institution}: {course.semester}
              {course.role && <> ({course.role}{course.collaborators && course.collaborators.length > 0 && (
                <>, with {course.collaborators.map((c, j) => (
                  <span key={j}>
                    {j > 0 && ", "}
                    {c.url ? (
                      <a href={c.url} target="_blank" rel="noopener noreferrer" className="text-text-light hover:text-accent-dark">{c.name}</a>
                    ) : c.name}
                  </span>
                ))}</>
              )})</>}
            </p>
          </div>
        ))}
      </div>

      <h2 className="text-xl font-semibold mb-4">Tutorials</h2>
      <div className="space-y-4">
        {tutorials.map((tutorial, i) => (
          <div key={i} className="bg-white rounded-lg p-5 shadow-sm">
            <h3 className="font-semibold text-lg">
              {tutorial.link ? (
                <a href={tutorial.link} target="_blank" rel="noopener noreferrer" className="text-text hover:text-accent">
                  {tutorial.name}
                </a>
              ) : (
                tutorial.name
              )}
            </h3>
            <p className="text-text-light text-sm mt-1">
              {tutorial.venue}
              {tutorial.slidesLink && (
                <>
                  {" "}
                  [<a href={tutorial.slidesLink} className="text-accent hover:text-accent-dark hover:underline">slides</a>]
                </>
              )}
            </p>
            <p className="text-text-light text-sm">{tutorial.collaborators}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
