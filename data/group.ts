export interface GroupMember {
  name: string;
  role?: string;
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

export interface ResearchTheme {
  title: string;
  description: string;
}

export interface JoiningBlock {
  heading: string;
  body: string;
}

export const labName = "The Landing";
export const labFullName = "The Language and Data Intelligence Group";

export const about: string[] = [
  "The Landing (<strong>The</strong> <strong>Lan</strong>guage and <strong>D</strong>ata <strong>In</strong>telligence <strong>G</strong>roup) in the Department of Computer Science at William &amp; Mary studies data-centric natural language processing (NLP). Broadly, we are interested in how data impacts model training and performance across all stages of model development and use. Our research centers on the idea of <strong>better data for better models</strong>.",
];

export const researchThemes: ResearchTheme[] = [
  {
    title: "Training Data Attribution",
    description:
      "Which training examples matter, and by how much? We develop methods for estimating the contribution of individual training data points, and apply them to data selection for pretraining and fine-tuning.",
  },
  {
    title: "Learning at Inference",
    description:
      "How does the data a model sees at inference time shape what it does? We study the impacts of in-context data, including which demonstrations help, how they are sampled and organized, and the biases they introduce.",
  },
  {
    title: "Evaluation and Analysis",
    description:
      "How do we know whether a model is actually good? We study the data generated from evaluation itself, including how framing and design choices impact human judgments, and how errors in model output are reported and analyzed.",
  },
];

export const currentMembers: GroupSection[] = [
  {
    heading: "Faculty",
    members: [
      {
        name: "Stephanie Schoch",
        role: "Assistant Professor",
        website: "/",
      },
    ],
  },
];

export const alumni: Alumnus[] = [];

export const joining: JoiningBlock[] = [
  {
    heading: "Prospective PhD students at W&amp;M",
    body:
      "I am recruiting 1&ndash;2 PhD students in the Fall 2026 and Spring 2027 semesters that are current (or incoming) William &amp; Mary students. If you are a current W&amp;M PhD student interested in NLP, please reach out by email with a short note about your background and what you would like to work on.",
  },
  {
    heading: "W&amp;M master's and undergraduate students",
    body:
      "If you are a current W&amp;M master's or undergraduate student who is interested in research, I am happy to chat. As I am new to W&amp;M, I am still in the process of learning what avenues are available for these students, so feel free to reach out and we can figure out together what makes sense.",
  },
  {
    heading: "Prospective students outside of W&amp;M",
    body:
      'Admissions decisions are made by a committee rather than myself directly. If you are interested in working with me as a PhD student, I would encourage you to apply to the <a href="https://cdsp.wm.edu/computerscience/graduate/" target="_blank" rel="noopener noreferrer">W&amp;M Computer Science PhD program</a>. Due to the high volume of emails I receive, I am unlikely to be able to reply to advising inquiries from students outside of W&amp;M.',
  },
];
