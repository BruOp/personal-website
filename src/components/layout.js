import React from "react";
import SideBar from "./sidebar";
import LayoutGrid from "./layout_grid";
import styled, { createGlobalStyle } from "styled-components";

import Footer from "./footer";
import { COLORS, media } from "../utils/style_helpers";

import "prismjs/plugins/line-numbers/prism-line-numbers.css";
import "./prism.css";
import { rhythm } from "../utils/typography";

const GlobalStyles = createGlobalStyle`
  color: ${COLORS.primary};

  a {
    color: ${COLORS.secondary}

  }

  a:visited {
    color: ${COLORS.secondary}
  }
`;

const MainContent = styled.main`
  border-top: 1px solid ${COLORS.lightest};
  padding-top: ${rhythm(1)};

  ${media.medium`
    padding: 0 0 0 ${rhythm(1)};
    border-left: 1px solid ${COLORS.lightest};
    border-top: 0
  `};
`;

class Layout extends React.Component {
  render() {
    const { children } = this.props;
    return (
      <>
        <GlobalStyles />
        <LayoutGrid>
          <SideBar />
          <MainContent>{children}</MainContent>
          <Footer />
        </LayoutGrid>
      </>
    );
  }
}

export default Layout;
