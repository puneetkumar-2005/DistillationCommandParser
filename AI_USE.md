# How I used AI in this project

I used one AI tool: Muse, Meta's AI assistant, through its chat interface. I used it through the whole project, but I ran every command myself on my own laptop, and every number in the report comes from a run on my laptop or from the CI pipeline, never from the assistant's machine.

What the assistant did for me: it explained the handout and the marking tables, drafted the code and the Kubernetes manifests, the worker and cleaning scripts, the service, the tests, the CI workflow, and the report and README text. It also helped me debug. When a command failed, I pasted the error and it suggested what to check next.

What I did myself: I typed and ran every command on my Dell G15, checked every output, and fixed the failures the runs exposed. The assistant's suggestions were wrong more than once, and I caught those by running them. The first rolling-update test failed with 10 HTTP 502 errors, and the preStop fix came from reading that failure, not from trusting a guess. The Makefile seed step failed because the release zip has a top-level folder, and I fixed the paths. The CI run failed on the transformers install, and splitting the installs fixed it. I also once committed an empty peak-memory file, and after that I read every evidence file back before citing it.

How I checked the AI's work: nothing went into the repo or the report without a run on my machine to back it. I ran the simulator on the Dell (6/6) instead of trusting a result from elsewhere, re-measured the cold start when the first timing was wrong, and threw away the first load-test run because it was warm-up. The report says where each number came from, and the GPU and CPU split is stated plainly: I used the laptop GPU (RTX 3050) for Task 1 labelling and Task 2 training because the CPU was measured at about 13-14 hours for labelling. Every service, load and latency measurement in Tasks 3, 4 and 7 is CPU-only, as the assignment expects.

That is the honest version: the AI did the drafting and the explaining, I did the running, the checking and the fixing, and the marks rest on what my laptop actually measured.
