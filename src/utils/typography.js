import Typography from "typography";
import FairyGates from "typography-theme-fairy-gates";

const typography = new Typography(FairyGates);

// Hot reload typography in development.
if (process.env.NODE_ENV !== `production`) {
  typography.injectStyles();
}

export default typography;
export const rhythm = typography.rhythm;
export const scale = typography.scale;
