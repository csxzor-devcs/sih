import { render, screen, act } from "@testing-library/react";
import ModelCardPage from "../app/(dashboard)/model-card/page";

describe("ModelCardPage", () => {
  it("renders the architecture section", async () => {
    await act(async () => { render(<>{await ModelCardPage()}</>); });
    expect(await screen.findByText(/Architecture/i)).toBeInTheDocument();
  });
  it("renders the metrics section", async () => {
    await act(async () => { render(<>{await ModelCardPage()}</>); });
    expect(await screen.findByRole("heading", { name: /Metrics/i, level: 2 })).toBeInTheDocument();
  });
  it("renders the limitations section", async () => {
    await act(async () => { render(<>{await ModelCardPage()}</>); });
    expect(await screen.findByText(/Limitations/i)).toBeInTheDocument();
  });
});
