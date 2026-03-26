export interface GroupMember {
  name: string;
  note?: string;
  website?: string;
}

export interface GroupSection {
  heading: string;
  members: GroupMember[];
}

export interface Alumnus {
  name: string;
  degree: string;
  position?: string;
  website?: string;
}

export const labName = "The Landing";
export const labFullName = "The Language and Data Intelligence Group";

export const currentMembers: GroupSection[] = [
  {
    heading: "Ph.D. Students",
    members: [
      { name: "Student Name", website: "#" },
      { name: "Student Name", website: "#" },
    ],
  },
  {
    heading: "M.S. Students",
    members: [
      { name: "Student Name", website: "#" },
    ],
  },
  {
    heading: "Undergraduate Students",
    members: [
      { name: "Student Name" },
    ],
  },
];

export const alumni: Alumnus[] = [
  { name: "Alumnus Name", degree: "M.S.", position: "Company/University Name" },
];
