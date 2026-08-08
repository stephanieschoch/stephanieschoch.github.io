export interface Collaborator {
  name: string;
  url?: string;
}

export interface Course {
  name: string;
  institution: string;
  semester: string;
  role: string;
  collaborators?: Collaborator[];
  link?: string;
}

export interface Tutorial {
  name: string;
  venue: string;
  collaborators: string;
  link?: string;
  slidesLink?: string;
}

export const courses: Course[] = [
  {
    name: "CSCI 680: Natural Language Processing",
    institution: "William & Mary",
    semester: "Fall 2026",
    role: "",
    // Course site lives in app/_nlp — the underscore keeps it out of routing
    // so it is not published. To launch it: rename the folder back to
    // app/nlp and uncomment the line below.
    // link: "/nlp",
  },
  {
    name: "CS 4710: Artificial Intelligence",
    institution: "University of Virginia",
    semester: "Spring 2024",
    role: "Co-Instructor",
    collaborators: [{ name: "Yangfeng Ji", url: "https://yangfengji.net/" }],
  },
];

export const tutorials: Tutorial[] = [
  {
    name: "Data Contribution Estimation for Machine Learning",
    venue: "NeurIPS 2023",
    collaborators: "Stephanie Schoch, Ruoxi Jia, Yangfeng Ji",
    link: "https://neurips.cc/virtual/2023/73959",
  },
];
