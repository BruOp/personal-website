import Typography from "typography";
import Theme from "typography-theme-us-web-design-standards";

const typography = new Typography(Theme);

if (process.env.NODE_ENV !== `production`) {
  typography.injectStyles();
}

export default typography;
export const rhythm = typography.rhythm;
export const scale = typography.scale;
