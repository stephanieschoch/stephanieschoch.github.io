import Image from "next/image";
import { profile } from "@/data/profile";

function ScholarIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

function LinkedInIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

function CVIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zM14 3.5L18.5 8H14V3.5zM6 20V4h6v6h6v10H6z" />
      <path d="M8 12h8v1.5H8zm0 3h8v1.5H8zm0 3h5v1.5H8z" />
    </svg>
  );
}

const iconMap: Record<string, React.FC> = {
  cv: CVIcon,
  scholar: ScholarIcon,
  x: XIcon,
  linkedin: LinkedInIcon,
};

export default function ProfileCard() {
  return (
    <div className="text-center">
      <Image
        src="/headshot.jpg"
        alt="Stephanie Schoch"
        width={176}
        height={176}
        className="w-44 h-44 rounded-full mx-auto mb-4 object-cover"
        priority
      />

      <h1 className="text-2xl font-bold text-text">{profile.name}</h1>
      <p className="text-accent font-medium mt-1">{profile.title}</p>
      <p className="text-text-light text-sm mt-1">
        {profile.department}
        <br />
        {profile.institution}
      </p>
      <p className="text-text-light text-sm mt-2">{profile.email}</p>

      {/* Social links */}
      <div className="flex justify-center gap-4 mt-4">
        {profile.socialLinks.map((link) => {
          const Icon = iconMap[link.icon];
          return (
            <a
              key={link.name}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:text-accent-dark transition-colors"
              title={link.name}
            >
              {Icon ? <Icon /> : link.name}
            </a>
          );
        })}
      </div>
    </div>
  );
}
