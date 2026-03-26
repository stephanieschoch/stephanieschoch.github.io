export interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  note?: string;
  link?: string;
}

export const publications: Publication[] = [
  {
    title: "The Good, the Bad, and the Debatable: A Survey on the Impacts of Data for In-Context Learning",
    authors: "Stephanie Schoch, Yangfeng Ji",
    venue: "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year: 2025,
    link: "https://aclanthology.org/2025.emnlp-main.1514/",
  },
  {
    title: "Monte Carlo Sampling for Analyzing In-Context Examples",
    authors: "Stephanie Schoch, Yangfeng Ji",
    venue: "Proceedings of the Sixth Workshop on Insights from Negative Results in NLP (NAACL Insights)",
    year: 2025,
    link: "https://aclanthology.org/2025.insights-1.7/",
  },
  {
    title: "In-Context Learning (and Unlearning) of Length Biases",
    authors: "Stephanie Schoch, Yangfeng Ji",
    venue: "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)",
    year: 2025,
    link: "https://aclanthology.org/2025.naacl-long.390/",
  },
  {
    title: "Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values",
    authors: "Stephanie Schoch, Ritwick Mishra, Yangfeng Ji",
    venue: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics Student Research Workshop (ACL SRW)",
    year: 2023,
    link: "https://aclanthology.org/2023.acl-srw.37/",
  },
  {
    title: "Barriers and enabling factors for error analysis in NLG research",
    authors: "Emiel van Miltenburg, Miruna Clinciu, Ondřej Dušek, Dimitra Gkatzia, Stephanie Inglis, Leo Leppänen, Saad Mahamood, Stephanie Schoch, Craig Thomson, Luou Wen",
    venue: "Northern European Journal of Language Technology (NEJLT)",
    year: 2023,
    link: "https://aclanthology.org/2023.nejlt-1.3/",
  },
  {
    title: "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification",
    authors: "Stephanie Schoch, Haifeng Xu, Yangfeng Ji",
    venue: "Proceedings of the 36th International Conference on Neural Information Processing Systems (NeurIPS)",
    year: 2022,
    link: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html",
  },
  {
    title: "Ask and you shall receive?: A LibAnswers FAQ usability study",
    authors: "Stephanie Schoch, Amanda VerMeulen",
    venue: "Weave: Journal of Library User Experience",
    year: 2022,
    link: "https://journals.publishing.umich.edu/weaveux/article/id/1023/",
  },
  {
    title: "Contextualizing Variation in Text Style Transfer Datasets",
    authors: "Stephanie Schoch, Wanyu Du, Yangfeng Ji",
    venue: "Proceedings of the 14th International Conference on Natural Language Generation (INLG)",
    year: 2021,
    link: "https://aclanthology.org/2021.inlg-1.22/",
  },
  {
    title: "Underreporting of errors in NLG output, and what to do about it",
    authors: "Emiel Van Miltenburg, Miruna-Adriana Clinciu, Ondřej Dušek, Dimitra Gkatzia, Stephanie Inglis, Leo Leppänen, Saad Mahamood, Emma Manning, Stephanie Schoch, Craig Thomson, Luou Wen",
    venue: "Proceedings of the 14th International Conference on Natural Language Generation (INLG)",
    year: 2021,
    note: "Outstanding Position Paper",
    link: "https://aclanthology.org/2021.inlg-1.14/",
  },
  {
    title: "'This is a Problem, Don't You Agree?' Framing and Bias in Human Evaluation for Natural Language Generation",
    authors: "Stephanie Schoch, Diyi Yang, Yangfeng Ji",
    venue: "Proceedings of the 1st Workshop on Evaluating NLG Evaluation (EvalNLGEval)",
    year: 2020,
    link: "https://aclanthology.org/2020.evalnlgeval-1.2/",
  },
];
