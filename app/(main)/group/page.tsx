// Hidden for now — uncomment this import block along with the header JSX below.
// import Image from "next/image";
// import { labName } from "@/data/group";

export const metadata = {
  title: "Group – Stephanie Schoch",
};

export default function GroupPage() {
  return (
    <div>
      {/* Hidden until the group page is ready — uncomment to restore the
          logo, name, and tagline (also uncomment the imports at the top).
      <div className="flex items-center gap-1 mb-8">
        <Image
          src="/landing_logo.png"
          alt="The Landing logo"
          width={256}
          height={256}
          className="w-36 h-36 object-contain -ml-9"
          quality={100}
          unoptimized
        />
        <div>
          <h1 className="text-2xl font-bold">{labName}</h1>
          <p className="text-text text-lg">
            The <strong>Lan</strong>guage and <strong>D</strong>ata <strong>In</strong>telligence <strong>G</strong>roup
          </p>
        </div>
      </div>
      */}

      <p className="text-text-light italic">More information coming soon...</p>
    </div>
  );
}
