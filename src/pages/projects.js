import React from "react";
import { StaticQuery, graphql } from "gatsby";
import styled from "styled-components";

import Layout from "../components/layout";

const ProjectWrapper = styled.div`
  
  ul > li {
    margin-bottom: 0.5em;
  }
`;

const Projects = () => {
  return (
    <StaticQuery
      query={projectsQuery}
      render={data => {
        const projects_markdown = data.markdownRemark;

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
              The two main projects I've worked on in my spare time are <a href="https://github.com/bruop/bae">bae</a>{" "}
              and <a href="https://github.com/bruop/zec">Zec</a>. Below are the "greatest hits" from those projects.
            </p>

            <ProjectWrapper>
              <div dangerouslySetInnerHTML={{ __html: projects_markdown.html }} />
            </ProjectWrapper>
          </Layout>
        );
      }}
    />
  );
};

const projectsQuery = graphql`
  query ProjectsQuery {
    markdownRemark(fields: { slug: { eq: "/portfolio/projects/" } }) {
      id
      html
    }
  }
`;

export default Projects;
