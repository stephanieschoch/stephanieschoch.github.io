import Image from "next/image";
import {
  labName,
  about,
  researchThemes,
  currentMembers,
  joining,
} from "@/data/group";
import { profile } from "@/data/profile";

export const metadata = {
  title: "Group – Stephanie Schoch",
};

// Wordmark and mark share one colour. Swap both together if this changes:
// landing-mark-navy.svg pairs with text-[#16324f].
const INK = "text-text";

export default function GroupPage() {
  return (
    <div>
      {/* Lockup: mark, dividing rule, wordmark over letterspaced tagline */}
      <div className="flex items-center gap-5 sm:gap-7 mb-10">
        <Image
          src="/landing-mark-black.svg"
          alt="The Landing logo: a Canada goose landing on water"
          width={815}
          height={754}
          className="w-24 sm:w-28 h-auto shrink-0"
          quality={100}
          priority
        />

        <div className="w-px self-stretch bg-gray-300" aria-hidden="true" />

        <div>
          <h1 className={`text-3xl sm:text-4xl font-bold leading-none ${INK}`}>
            {labName}
          </h1>
          <p className="mt-2 text-[0.7rem] sm:text-xs uppercase tracking-[0.14em] text-accent">
            <strong className="font-bold">The</strong>{" "}
            <strong className="font-bold">Lan</strong>guage and{" "}
            <strong className="font-bold">D</strong>ata{" "}
            <strong className="font-bold">In</strong>telligence{" "}
            <strong className="font-bold">G</strong>roup
          </p>
        </div>
      </div>

      {about.map((paragraph, i) => (
        <p
          key={i}
          className="mb-4 leading-relaxed [&_a]:text-accent [&_a]:hover:text-accent-dark"
          dangerouslySetInnerHTML={{ __html: paragraph }}
        />
      ))}

      <h2 className="text-2xl font-bold mt-10 mb-4">Research</h2>
      <div className="space-y-4">
        {researchThemes.map((theme) => (
          <div key={theme.title}>
            <h3 className="font-semibold">{theme.title}</h3>
            <p className="text-sm leading-relaxed text-text-light">
              {theme.description}
            </p>
          </div>
        ))}
      </div>

      <h2 className="text-2xl font-bold mt-10 mb-4">People</h2>
      {currentMembers.map((section) => (
        <div key={section.heading} className="mb-4">
          <h3 className="font-semibold mb-2">{section.heading}</h3>
          <ul className="text-sm space-y-1">
            {section.members.map((m) => (
              <li key={m.name}>
                {m.website ? (
                  <a
                    href={m.website}
                    className="text-accent hover:text-accent-dark"
                  >
                    {m.name}
                  </a>
                ) : (
                  m.name
                )}
                {m.role && <span className="text-text-light"> — {m.role}</span>}
              </li>
            ))}
          </ul>
        </div>
      ))}
      <p className="text-sm text-text-light italic">
        The group is newly formed. See below if you are interested in joining.
      </p>

      <h2 className="text-2xl font-bold mt-10 mb-4">Joining the Group</h2>
      <div className="bg-[#eef4fb] rounded-lg p-6 space-y-5">
        {joining.map((block) => (
          <div key={block.heading}>
            <h3
              className="font-semibold mb-1"
              dangerouslySetInnerHTML={{ __html: block.heading }}
            />
            <p
              className="text-sm leading-relaxed [&_a]:text-accent [&_a]:hover:text-accent-dark [&_a]:underline"
              dangerouslySetInnerHTML={{ __html: block.body }}
            />
          </div>
        ))}
        <p className="text-sm text-text-light pt-1">
          Email: {profile.email}
        </p>
      </div>
    </div>
  );
}
