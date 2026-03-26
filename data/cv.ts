export interface CVEntry {
  title: string;
  subtitle?: string;
  date: string;
  details?: string;
}

export interface CVSection {
  heading: string;
  entries: CVEntry[];
}

export const cv: CVSection[] = [
  {
    heading: "Education",
    entries: [
      {
        title: "Ph.D. in Computer Science",
        subtitle: "University of Virginia",
        date: "2019 – 2024",
        details: "Dissertation: Placeholder dissertation title",
      },
      {
        title: "B.S. in Computer Science",
        subtitle: "University Name",
        date: "2015 – 2019",
      },
    ],
  },
  {
    heading: "Experience",
    entries: [
      {
        title: "Assistant Professor",
        subtitle: "University Name, Department of Computer Science",
        date: "2024 – Present",
      },
      {
        title: "Research Intern",
        subtitle: "Research Lab Name",
        date: "Summer 2023",
        details: "Worked on large language model evaluation.",
      },
    ],
  },
  {
    heading: "Awards & Honors",
    entries: [
      {
        title: "Outstanding Dissertation Award",
        subtitle: "University of Virginia",
        date: "2024",
      },
      {
        title: "Best Paper Award",
        subtitle: "Conference Name",
        date: "2023",
      },
    ],
  },
  {
    heading: "Service",
    entries: [
      {
        title: "Program Committee Member",
        subtitle: "ACL, EMNLP, NAACL",
        date: "2023 – Present",
      },
      {
        title: "Reviewer",
        subtitle: "Computational Linguistics, TACL",
        date: "2022 – Present",
      },
    ],
  },
];
