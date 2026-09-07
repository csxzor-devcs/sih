import { render, screen } from "@testing-library/react";
import SimulatorPage from "../app/(dashboard)/simulator/page";

it("renders the controls panel", () => {
  render(<SimulatorPage />);
  expect(screen.getByText(/What-If Simulator/i)).toBeInTheDocument();
});

it("renders a submit button", () => {
  render(<SimulatorPage />);
  expect(screen.getByRole("button", { name: /Forecast/i })).toBeInTheDocument();
});
