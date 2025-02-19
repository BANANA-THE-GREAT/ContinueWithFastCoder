import GenerateTerminalCommand from "./cmd";
import CommitMessageCommand from "./commit";
import DisableAccelerationMethodCommand from "./disable";
import DraftIssueCommand from "./draftIssue";
import EnableAccelerationMethodCommand from "./enable";
import HttpSlashCommand from "./http";
import OnboardSlashCommand from "./onboard";
import RenewRepoDatastoreCommand from "./renew";
import ReviewMessageCommand from "./review";
import ShareSlashCommand from "./share";
import EnableGuiMethodCommand from "./enableGui";
import DisableGuiMethodCommand from "./disableGui";

export default [
  DraftIssueCommand,
  ShareSlashCommand,
  GenerateTerminalCommand,
  HttpSlashCommand,
  CommitMessageCommand,
  ReviewMessageCommand,
  OnboardSlashCommand,
  RenewRepoDatastoreCommand,
  EnableAccelerationMethodCommand,
  DisableAccelerationMethodCommand,
  EnableGuiMethodCommand,
  DisableGuiMethodCommand,
];
