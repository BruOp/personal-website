import React from "react";
import { StaticQuery, graphql } from "gatsby";
import Image from "gatsby-image";
import styled from "styled-components";
import { rhythm, scale } from "../utils/typography";

import NavList from "./nav_list";
import { media } from "../utils/style_helpers";
import SocialLinks from "./social";

const ProfilePic = styled(Image)``;

const SideBarBlurb = styled.div`
  h2 {
    ${scale(0.2)};
    margin-top: 0;
  }

  p {
    ${scale(-0.1)}
    color: #666666;
    display: none;
  }

  ${SocialLinks} {
    display: none;
  }

  ${media.medium`

    p {
      display: block;
    }

    ${NavList} {
      margin-bottom: ${rhythm(1.0)};
    }

    ${SocialLinks} {
      display: block;
    }
  `}
`;

const SideBarWrapper = styled.div`
  display: grid;

  grid-gap: ${rhythm(0.5)};
  grid-template-rows: auto;
  grid-template-columns: auto 1fr;
  grid-template-areas: "pic text";

  align-content: center;

  ${media.medium`
    align-content: initial;
    grid-template-columns: 1fr;
    grid-template-rows: auto auto;
    grid-template-areas:
      "pic"
      "text";
  `}

  ${ProfilePic} {
    justify-self: center;
    grid-area: pic;
  }

  ${SideBarBlurb} {
    grid-area: text;
  }
`;

let SideBar = function({ className }) {
  const author = "Bruno Opsenica";
  return (
    <StaticQuery
      query={sideBarQuery}
      render={data => {
        return (
          <div className={className}>
            <SideBarWrapper>
              <ProfilePic
                fixed={data.avatar.childImageSharp.fixed}
                alt={author}
                imgStyle={{
                  borderRadius: `50%`,
                }}
              />
              <SideBarBlurb>
                <h2>{author}'s Blog</h2>
                <p>This blog chronicles the topics in real time rendering I'm learning about.</p>
                <NavList />
                <SocialLinks />
              </SideBarBlurb>
            </SideBarWrapper>
          </div>
        );
      }}
    />
  );
};

const sideBarQuery = graphql`
  query SideBarQuery {
    avatar: file(absolutePath: { regex: "/profile-pic.jpg/" }) {
      childImageSharp {
        fixed(width: 80, height: 80, quality: 90) {
          ...GatsbyImageSharpFixed
        }
      }
    }
  }
`;

SideBar = styled(SideBar)``;

export default SideBar;
