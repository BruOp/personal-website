import styled from "styled-components";
import { COLORS } from "../utils/style_helpers";
import { rhythm } from "../utils/typography";

const BlogPostDate = styled.small`
  color: ${COLORS.tertiary};
  font-weight: 700;
  margin-bottom: ${rhythm(0.5)};
  display: block;
`;
export default BlogPostDate;
