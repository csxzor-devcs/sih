import { render, screen, act } from "@testing-library/react";
import BreakdownPage from "../app/(dashboard)/metrics/breakdown/page";

describe("BreakdownPage", () => {
  it("renders per-class AUROC table", async () => {
    await act(async () => { render(<>{await BreakdownPage()}</>); });
    expect(await screen.findByText(/Per-Class AUROC/i)).toBeInTheDocument();
  });
});
