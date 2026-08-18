export const metadata = {
  title: "Project – Natural Language Processing",
};

export default function ProjectPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Project</h1>

      <div className="space-y-8">
        {/* Project Pitch */}
        <section>
          <h2 className="text-xl font-semibold mb-2">Project Pitch (5%)</h2>
          <p>
            Each student will pitch an idea for an NLP project to the class via
            a 5-minute presentation. This is aimed to assist in the team
            formation process. The pitch should address 1) what the problem is
            that will be solved, 2) why the problem matters, and 3) a
            preliminary idea about why this project is feasible.
          </p>
        </section>

        {/* Project Team Formation */}
        <section>
          <h2 className="text-xl font-semibold mb-2">Project Team Formation</h2>
          <p>
            After project pitches, students will be expected to form teams of
            2-3 to work on an agreed upon project idea.
          </p>
        </section>

        {/* Project Proposal */}
        <section>
          <h2 className="text-xl font-semibold mb-2">Project Proposal (10%)</h2>
          <p className="mb-3">
            Teams will submit a 1-2 page project proposal (note: references do
            not count towards limit). The proposal is expected to demonstrate
            that the teams have thought about their problem, understand how it
            fits within the related literature, and have a clear and feasible
            plan to move forward (e.g. considered data and compute resources).
          </p>
          <h3 className="text-lg font-medium mb-2 italic">
            Proposal Requirements:
          </h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <strong>Task Definition:</strong> Define the problem and problem
              motivation. Where relevant, include example inputs and expected
              outputs.
            </li>
            <li>
              <strong>Related Work:</strong> Outline the connections between the
              proposed problem and the related literature.
              <ul className="list-disc pl-6 mt-1">
                <li>
                  Sources to Look for Related Literature: ACL Anthology,
                  OpenReview (NeurIPS, ICLR, etc.)
                </li>
              </ul>
            </li>
            <li>
              <strong>Hypothesis:</strong> Statement of your hypothesis and what
              criteria you will use to evaluate its validity.
            </li>
            <li>
              <strong>Preliminary Plan:</strong> A brief description of your
              preliminary ideas for your approach to test your hypothesis. This
              may include potential models and baselines, datasets, evaluation
              strategies, etc.
            </li>
          </ul>
        </section>

        {/* Project Progress Report */}
        <section>
          <h2 className="text-xl font-semibold mb-2">
            Project Progress Report (10%)
          </h2>
          <p className="mb-3">
            Teams will submit a ~3 page progress report. This will outline any
            changes that were made in response to proposal feedback and should
            demonstrate that non-trivial progress has been made.
          </p>
          <h3 className="text-lg font-medium mb-2 italic">
            Progress Report Requirements:
          </h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <strong>Task Definition:</strong> Define the problem and problem
              motivation. Where relevant, include example inputs and expected
              outputs.
            </li>
            <li>
              <strong>Dataset Details:</strong> At this point, the dataset
              should be finalized.
            </li>
            <li>
              <strong>Initial Results:</strong> This should be considered a
              proof of concept (i.e. a motivating result or analysis).
            </li>
            <li>
              <strong>Finalized Plan:</strong> A concrete plan of what will be
              accomplished (and how) for the final report.
            </li>
          </ul>
        </section>

        {/* Project Final Presentation */}
        <section>
          <h2 className="text-xl font-semibold mb-2">
            Project Final Presentation (10%)
          </h2>
          <p>
            Each team will give a 15-20 minute presentation, with 5-10 minutes
            for QA.
          </p>
        </section>

        {/* Project Final Report */}
        <section>
          <h2 className="text-xl font-semibold mb-2">Project Final Report (15%)</h2>
          <p className="mb-3">
            Teams will submit an 8 page (excluding references and appendix)
            final report.
          </p>
          <h3 className="text-lg font-medium mb-2 italic">
            Final Report Requirements:
          </h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <strong>Abstract:</strong> What the problem is, why it matters,
              what the approach is, and main takeaways from the results.
            </li>
            <li>
              <strong>Introduction:</strong> Introduce and motivate the problem.
              Brief description of what is accomplished in your project.
            </li>
            <li>
              <strong>Related Work:</strong> A brief description of the related
              work, outlining the connections and differences between your work
              and existing works.
            </li>
            <li>
              <strong>Method:</strong> A detailed description of your method.
            </li>
            <li>
              <strong>Experiment Setup:</strong> A description of what models,
              hyperparameters, baselines, tools, datasets, etc. were used.
            </li>
            <li>
              <strong>Results:</strong> Quantitative results with descriptive
              analysis. Results should include tables or figures, which are
              referenced in the text.
            </li>
            <li>
              <strong>Discussion:</strong> Additional discussion of how your
              results relate to your initial hypothesis, a discussion of future
              work, and a brief discussion of limitations.
            </li>
            <li>
              <strong>References:</strong> Following ACL formatting.
            </li>
            <li>
              <strong>Appendix:</strong> If needed for additional details or
              results.
            </li>
            <li>
              <strong>Other Requirements:</strong> The report should follow the
              structure of a *ACL publication. It should include a link to a
              Github repo with your code and data.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
