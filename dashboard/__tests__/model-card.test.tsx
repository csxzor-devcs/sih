import { render, screen } from "@testing-library/react";
import ModelCardPage from "../app/(dashboard)/model-card/page";

describe("ModelCardPage", () => {
  it("renders the architecture section", () => {
    render(<ModelCardPage />);
    expect(screen.getByText(/Architecture/i)).toBeInTheDocument();
  });
  it("renders the metrics section", () => {
    render(<ModelCardPage />);
    expect(screen.getByText(/Metrics/i)).toBeInTheDocument();
  });
  it("renders the limitations section", () => {
    render(<ModelCardPage />);
    expect(screen.getByText(/Limitations/i)).toBeInTheDocument();
  });
});
