import { css } from "styled-components";

export const COLORS = {
  primary: "#444444",
  secondary: "#1ca086",
  tertiary: "#EF9803",
  lighter: "#666666",
  lightest: "#dddddd",
};

export const sizes = {
  large: 900,
  medium: 768,
};

// Iterate through the sizes and create a media template
export const media = Object.keys(sizes).reduce((acc, label) => {
  acc[label] = (...args) => css`
    @media (min-width: ${sizes[label] / 16}em) {
      ${css(...args)}
    }
  `;

  return acc;
}, {});
