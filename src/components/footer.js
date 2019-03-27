import React from "react";
import styled from "styled-components";
import { scale } from "../utils/typography";
import { COLORS } from "../utils/style_helpers";
import SocialLinks from "./social";

let Footer = ({ className }) => {
  return (
    <footer className={className}>
      <SocialLinks />
      <div>©{new Date().getFullYear()}, Bruno Opsenica</div>
    </footer>
  );
};

Footer = styled(Footer)`
  /* justify-content: space-between; */
  ${scale(-0.3)}
  color: ${COLORS.lighter};
  /* display: flex; */
  /* align-items: center; */
  text-align: right;
`;

export default Footer;
