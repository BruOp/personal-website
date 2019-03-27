import styled from "styled-components";
import SideBar from "./sidebar";
import { rhythm } from "../utils/typography";
import { media } from "../utils/style_helpers";

const LayoutGrid = styled.div`
  display: grid;
  max-width: 1024px;
  margin: auto;
  min-height: 100vh;
  padding: ${rhythm(0.75)} ${rhythm(1)} 0;
  grid-gap: ${rhythm(1)};

  grid-template-columns: 1fr;
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "sidebar"
    "main-content"
    "footer";

  ${media.medium`
    grid-template-columns: 220px 1fr;
    grid-template-rows: 24px 1fr auto;
    grid-template-areas:
      "header header"
      "sidebar main-content"
      "footer footer";
  `}

  & > footer {
    grid-area: footer;
  }

  & > ${SideBar} {
    grid-area: sidebar;
  }

  & > main {
    grid-area: main-content;
  }
`;

export default LayoutGrid;
