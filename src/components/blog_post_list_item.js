import React from "react";
import { Link } from "gatsby";
import styled from "styled-components";
import { rhythm } from "../utils/typography";
import BlogPostDate from "./blog_post_date";

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
      <BlogPostDate>{post.frontmatter.date}</BlogPostDate>
      <PostBlurb
        dangerouslySetInnerHTML={{
          __html: post.frontmatter.description || post.excerpt,
        }}
      />
    </div>
  );
};

export default BlogPostListItem;
