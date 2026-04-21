# Handover for Kaarthik: Data Analysis Instructions

**Primary Analysis Target:**
The main task is to analyze the top of the recorded bursts to show traveling waves propagating across the grid. The analysis should also measure the propagation speed of these bursts.
**Noise and Artifacts:**

- The data contains high-frequency noise from equipment that should be non-linearly filtered out.
- There are occasional, large, and obvious breathing-related movement artifacts that need to be cut out from the analysis.
- A key limitation is that the data is chopped under 50 Hz, making it difficult to distinguish between muscle noise and other high-frequency sources.
  **Key Observations:**
- The recordings show a consistent burst suppression activity. The leading edge of this activity appears to originate from the implants on the right side, which seem to be activating the whole area.
- By extracting the "strange first types" of bursts, you can identify the exact location of the irritation.
- The activity is consistently observed on the "top half" of the grid. It is good if the top half is the right, and bad if it's reversed.
  **Recording Parameters:**
- **Filters:** Wide-band recording with a 5 Hz high-pass and 50 Hz low-pass filter.
- **Denoising:** Gaussian denoising was applied.
- **Anesthesia:** The level was raised from 2% to 2.25% during the session.
