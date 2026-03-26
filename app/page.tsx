import ProfileCard from "@/components/ProfileCard";
import { profile } from "@/data/profile";
import { updates } from "@/data/updates";

export default function Home() {
  return (
    <div className="flex flex-col md:flex-row gap-10 items-start">
      <aside className="w-full md:w-64 shrink-0">
        <ProfileCard />
      </aside>
      <section className="flex-1">
        <h2 className="text-2xl font-bold mb-4">About Me</h2>
        {profile.about.map((paragraph, i) => (
          <p key={i} className="mb-4 leading-relaxed [&_a]:text-text [&_a]:hover:text-accent" dangerouslySetInnerHTML={{ __html: paragraph }} />
        ))}

        <h2 className="text-2xl font-bold mb-4 mt-8">Updates</h2>
        <div className="bg-[#eef4fb] rounded-lg p-4 max-h-72 overflow-y-auto">
          <ul className="space-y-2 text-sm">
            {updates.map((update, i) => (
              <li key={i} className="[&_a]:text-accent [&_a]:hover:text-accent-dark">
                <span className="font-semibold">[{update.date}]</span>{" "}
                <span dangerouslySetInnerHTML={{ __html: update.text }} />
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}
