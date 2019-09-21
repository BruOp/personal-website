import React from "react";
import { StaticQuery, graphql } from "gatsby";
import Image from "gatsby-image";
import styled from "styled-components";
import { media } from "../utils/style_helpers";

import Layout from "../components/layout";

const ProjectList = styled.ul`
  list-style: none;
  margin: 0;
  display: flex;
  width: 100%;
  height: auto;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
`;

const Link = styled.a`
  display: block;
  transition: transform 200ms;
  text-decoration: none;

  .gatsby-image-wrapper {
    transition: opacity 200ms;
    opacity: 1;
  }

  :hover {
    transform: scale(1.1);
    z-index: 1;

    .gatsby-image-wrapper {
      opacity: 0.6;
    }
  }
`;

const ProjectItem = styled.li`
  border-radius: 100%;
  width: 100%;
  margin: 10px;

  ${media.medium`
    width: 45%
  `}

  ${media.large`
    width: 40%
  `}

  h4 {
    margin: 5px 0 0;
    text-align: center;
  }
`;

const Project = ({ link, imgData, title }) => {
  return (
    <ProjectItem>
      <Link href={link}>
        <Image fluid={imgData.childImageSharp.fluid} />
        <h4>{title}</h4>
      </Link>
    </ProjectItem>
  );
};

const StyledProject = styled(Project)``;

const Projects = () => {
  return (
    <StaticQuery
      query={projectsQuery}
      render={data => {
        return (
          <Layout>
            <h1>Portfolio</h1>
            <p>
              The projects I've worked on can basically be broken down into two categories: web development projects and
              personal computer graphics projects. Occasionally there is overlap, as with{" "}
              <a href="https://echoes.cbc.ca/">Echoes</a>, but for the most part I've only documented my computer
              graphics projects on this page.
            </p>
            <p>
              For the past few months I've been working on writing small, mostly self-contained examples for different
              rendering techniques. They are open souced on GitHub in{" "}
              <a href="https://github.com/bruop/bae">this repo</a>.
            </p>
            <p></p>
            <ProjectList>
              <StyledProject
                link="https://github.com/bruop/bae#tone-mapping"
                imgData={data.tone_mapping}
                title="HDR Tone Mapping"
              />
              <StyledProject
                link="https://github.com/bruop/bae#forward-vs-deferred-rendering"
                imgData={data.forward}
                title="Forward Rendering of Sponza"
              />
              <StyledProject
                link="https://github.com/bruop/bae#deferred"
                imgData={data.deferred}
                title="Deferred Rendering of Sponza"
              />
              <StyledProject
                link="https://github.com/bruop/bae#physically-based-image-based-lighting"
                imgData={data.ibl}
                title="Image Based Lighting"
              />
              <StyledProject
                link="https://github.com/bruop/bae#cascaded-shadow-maps"
                imgData={data.shadows}
                title="Cascaded Shadow Maps"
              />
              <StyledProject link="https://echoes.cbc.ca/" imgData={data.echoes} title="Echoes" />
            </ProjectList>
          </Layout>
        );
      }}
    />
  );
};

const projectsQuery = graphql`
  query ProjectsQuery {
    echoes: file(absolutePath: { regex: "/echoes.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
    tone_mapping: file(absolutePath: { regex: "/01-tonemapping.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
    forward: file(absolutePath: { regex: "/02-forward-rendering.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
    deferred: file(absolutePath: { regex: "/03-deferred.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
    ibl: file(absolutePath: { regex: "/04-pbr-ibl.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
    shadows: file(absolutePath: { regex: "/05-shadow-mapping.png/" }) {
      childImageSharp {
        fluid {
          ...GatsbyImageSharpFluid
        }
      }
    }
  }
`;

export default Projects;
