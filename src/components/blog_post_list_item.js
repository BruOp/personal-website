import React from "react";
import { Link } from "gatsby";
import styled from "styled-components";
import { COLORS } from "../utils/style_helpers";
import { rhythm } from "../utils/typography";

const PostDate = styled.small`
  color: ${COLORS.tertiary};
  font-weight: 700;
  margin-bottom: ${rhythm(0.5)};
  display: block;
`;

const PostTitle = styled.h3`
  margin-bottom: 0;
`;

const PostBlurb = styled.p`
  margin-bottom: ${rhythm(1.2)};
`;

let BlogPostListItem = ({ post, className }) => {
  const title = post.frontmatter.title || post.fields.slug;
  return (
    <div className={className}>
      <PostTitle>
        <Link to={post.fields.slug}>{title}</Link>
      </PostTitle>
      <PostDate>{post.frontmatter.date}</PostDate>
      <PostBlurb
        dangerouslySetInnerHTML={{
          __html: post.frontmatter.description || post.excerpt,
        }}
      />
    </div>
  );
};

export default BlogPostListItem;
