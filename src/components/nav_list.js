import React from "react";
import styled from "styled-components";
import { Link } from "gatsby";
import { media } from "../utils/style_helpers";
import { rhythm } from "../utils/typography";

const links = [
  {
    to: "/",
    text: "Blog Posts",
  },
  {
    to: "/about_me",
    text: "About Me",
  },
];

let NavList = ({ className }) => {
  return (
    <nav className={className}>
      <ul>
        {links.map((link, i) => (
          <li key={i}>
            <Link to={link.to}>{link.text}</Link>
          </li>
        ))}
      </ul>
    </nav>
  );
};

NavList = styled(NavList)`
  ul {
    list-style: none;
    padding-left: 0;
    margin-left: 0;
    margin: 0;

    ${media.medium`
      margin: initial;
    `}
  }

  li {
    display: inline-block;
    margin: 0 ${rhythm(0.5)} 0 0;

    ${media.medium`
      display: block;
      margin: initial;
    `}
  }
`;

export default NavList;
