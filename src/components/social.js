import React from "react";
import styled from "styled-components";
import Icon, { ICONS } from "./icon";
import { rhythm } from "../utils/typography";
import { COLORS } from "../utils/style_helpers";

const LINKS = [
  {
    icon: ICONS.GITHUB,
    to: "https://github.com/BruOp",
    alt: "Github Profile",
  },
  {
    icon: ICONS.TWITTER,
    to: "https://twitter.com/BruOps",
    alt: "@BruOps twitter profile",
  },
  {
    icon: ICONS.EMAIL,
    to: "mailto:bruno.opsenica@gmail.com",
    alt: "Email",
  },
];

const SocialLink = styled.li`
  display: inline-block;
  height: ${rhythm(1.4)};
  width: ${rhythm(1.4)};
  padding: ${rhythm(0.3)};
  margin-right: 8px;

  border-radius: 100%;
  border: 1px solid ${COLORS.lightest};
  box-sizing: border-box;
  transition: fill 500ms, border 500ms;

  &:hover {
    border: 1px solid ${COLORS.tertiary};
    fill: ${COLORS.tertiary};
  }
`;

let SocialLinks = ({ className }) => {
  return (
    <ul className={className}>
      {LINKS.map((link, i) => {
        return (
          <SocialLink key={i}>
            <a href={link.to}>
              <Icon icon={link.icon} />
            </a>
          </SocialLink>
        );
      })}
    </ul>
  );
};

SocialLinks = styled(SocialLinks)`
  list-style: none;
  margin: 0;
`;

export default SocialLinks;
